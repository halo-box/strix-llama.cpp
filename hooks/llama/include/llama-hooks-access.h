#pragma once

// Access to private members of upstream classes without a friend line in upstream.
// An explicit template instantiation can name a private member (C++ [temp.spec]): the instantiation stores a pointer to
// the member, a friend function returns it.
//
//   LLAMA_HOOK_ACCESS(llama_context_sched, llama_context, sched, ggml_backend_sched_ptr);
//   ggml_backend_sched_ptr & s = llama_hook_get(ctx, llama_context_sched{});
//
// Not for reference members (a pointer to a reference member does not exist): use a public accessor of the class.
// A renamed or removed member breaks the build, it cannot conflict.

template <typename Tag, typename Tag::type Member>
struct llama_hook_access_impl {
    friend typename Tag::type llama_hook_member(Tag) { return Member; }
};

#define LLAMA_HOOK_ACCESS(tag, Class, member, Type)                       \
    struct tag {                                                          \
        typedef Type Class::*type;                                        \
        friend type llama_hook_member(tag);                               \
    };                                                                    \
    template struct llama_hook_access_impl<tag, &Class::member>

template <typename Class, typename Tag>
auto & llama_hook_get(Class & obj, Tag tag) {
    return obj.*llama_hook_member(tag);
}

template <typename Class, typename Tag>
const auto & llama_hook_get(const Class & obj, Tag tag) {
    return obj.*llama_hook_member(tag);
}
